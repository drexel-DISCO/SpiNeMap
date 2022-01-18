import numpy as np
import sys
from lxml import etree

snn_name  = sys.argv[1]

connection_file = snn_name
output_file     = snn_name.replace('txt','xml')

#connection_file = sys.argv[1]
#output_file = sys.argv[2]

#class definitions
class actor:
    name      = 'actor'
    actor_id  = -1
    in_ports  = []
    out_ports = []
    in_rates  = []
    out_rates = []

    def __init__(self,aid):
        self.name      = 'actor_' + str(aid)
        self.actor_id  = aid
        self.in_ports  = []
        self.out_ports = []
        self.in_rates  = []
        self.out_rates = []
        #print 'AD: ',self.actor_id

    def add_in_ports(self,tokens):
        n_in_ports   = len(self.in_ports)
        in_port_name = 'in_port_' + str(n_in_ports)
        self.in_ports.append(in_port_name)
        self.in_rates.append(tokens)
        return in_port_name

    def add_out_ports(self,tokens):
        n_out_ports    = len(self.out_ports)
        out_port_name  = 'out_port_' + str(n_out_ports)
        self.out_ports.append(out_port_name)
        self.out_rates.append(tokens)
        return out_port_name

    def get_actor_id(self):
        return self.actor_id

    def get_name(self):
        return self.name

    def get_in_ports(self):
        return self.in_ports

    def get_out_ports(self):
        return self.out_ports

    def get_in_rates(self):
        return self.in_rates

    def get_out_rates(self):
        return self.out_rates

class connection:
    name        = 'connection'
    source      = -1
    destination = -1
    source_port_name = 'src_port'
    destination_port_name = 'dst_port'
    initial_token = 0

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_source(self,source):
        self.source = source

    def set_destination(self,destination):
        self.destination = destination

    def set_source_port(self,source_port):
        self.source_port_name = source_port

    def set_destination_port(self,destination_port):
        self.destination_port_name = destination_port

    def set_initial_token(self,token):
        self.initial_token = token

    def get_source(self):
        return self.source

    def get_destination(self):
        return self.destination

    def get_source_port(self):
        return self.source_port_name

    def get_destination_port(self):
        return self.destination_port_name

    def get_initial_token(self):
        return self.initial_token

#class definitions end here

#function definitions here
def get_actor_coord(actor_list,aid):
    coord = -1
    if len(actor_list) > 0:
        for i in range(len(actor_list)):
            if actor_list[i].get_actor_id() == aid:
                coord = i
                break
    return coord

#function definitions end here

#create dynamic data structrure for actors and destinations
actors = []
connections = []

count = 0
#open the file for reading
f = open(connection_file,"r")
for line in f:
    line_array  = line.strip().split()
    source      = int(line_array[0])
    destination = int(line_array[1])
    tokens      = int(line_array[2])

    #Fill for the source actor
    coord_src_actor = get_actor_coord(actors,source)
    if coord_src_actor < 0: #Actor not seen before
        a = actor(source)
        src_port_name = a.add_out_ports(tokens)
        actors.append(a)
    else:                   #Actors previously seen
        src_port_name = actors[coord_src_actor].add_out_ports(tokens)
    #Fill for the destination actor
    coord_dst_actor = get_actor_coord(actors,destination)
    if coord_dst_actor < 0: #Actor not seen before
        a = actor(destination)
        dst_port_name = a.add_in_ports(tokens)
        actors.append(a)
    else:                   #Actors previously seen
        dst_port_name = actors[coord_dst_actor].add_in_ports(tokens)
       
    #print [coord_src_actor,coord_dst_actor]    
    
    #Fill for the connections
    connection_name        = 'connection_' + str(source) + '_' + str(destination)
    c = connection(connection_name)     #Create an object for the connection and fill the fields
    c.set_source(source)
    c.set_destination(destination)
    c.set_source_port(src_port_name)
    c.set_destination_port(dst_port_name)
    c.set_initial_token(tokens)

    connections.append(c)               #Insert the connection object into the connections list 


f.close()


#XML generation here
root = etree.Element('sdf3',
        type="sdf",
        version="1.0")
        #xsinoNamespaceSchemaLocation="http://www.es.ele.tue.nl/sdf3/xsd/sdf3-sdf.xsd")

child1 = etree.Element('applicationGraph',
        name=snn_name)
root.append(child1)

child2 = etree.Element('sdf',
        name=snn_name,
        type="SNN2SDF")
child1.append(child2)

for i in range(len(actors)):
    child3 = etree.Element('actor',
        name=actors[i].get_name(),
        type="crossbar")
    child2.append(child3)

    in_ports = actors[i].get_in_ports()
    in_rates = actors[i].get_in_rates()
    for j in range(len(in_ports)):
        child4 = etree.Element('port',
            name=in_ports[j],
            type="in",
            rate=str(in_rates[j]))
        child3.append(child4)

    out_ports = actors[i].get_out_ports()
    out_rates = actors[i].get_out_rates()
    for j in range(len(out_ports)):
        child4 = etree.Element('port',
            name=out_ports[j],
            type="out",
            rate=str(out_rates[j]))
        child3.append(child4)

for i in range(len(connections)):
    srcActor = 'actor_'+str(connections[i].get_source())
    dstActor = 'actor_'+str(connections[i].get_destination())
    child3 = etree.Element('channel',
        name=connections[i].get_name(),
        srcActor=srcActor,
        srcPort=connections[i].get_source_port(),
        dstActor=dstActor,
        dstPort=connections[i].get_destination_port(),
        initialTokens=str(connections[i].get_initial_token()))
    child2.append(child3)

child2 = etree.Element('sdfProperties')
child1.append(child2)

for i in range(len(actors)):
    child3 = etree.Element('actorProperties',
            actor=actors[i].get_name())
    child2.append(child3)
    
    child4 = etree.Element('processor',
            type="crossbar",
            default="true")
    child3.append(child4)

    child5 = etree.Element('executionTime',
            time="4")
    child4.append(child5)

    child5 = etree.Element('memory')
    child4.append(child5)

    child6 = etree.Element('stateSize',
            max="2")
    child5.append(child6)

for i in range(len(connections)):
    child3 = etree.Element('channelProperties',
            channel=connections[i].get_name())
    child2.append(child3)

    child4 = etree.Element('tokenSize',
            sz="1")
    child3.append(child4)

child3 = etree.Element('graphProperties')
child2.append(child3)

child4 = etree.Element('timeConstraints')
child3.append(child4)

child5 = etree.Element('throughput')
child5.text = '0.00000003'
child4.append(child5)

t = etree.ElementTree(root)
t.write(output_file, pretty_print=True, xml_declaration=True,   encoding="utf-8")
