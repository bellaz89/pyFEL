import re
import toml
import json
import sys
import ast

def parse_lattice_native(fname):
    '''
        Parse native Genesis1.3v4 lattice files
    '''

    parsed_lattice = dict()
    element_re = re.compile("^(\w+):(\w+)=\{(.*)\}$")
    elements = open(fname, "r").read().replace(" ","").replace("\n","").split(";")
    
    for element in elements:         
        if element.replace(" ","") == "":
            continue

        matched = element_re.match(element.replace(" ",""))
        if matched:
            parsed_lattice[matched.group(1)] = dict()
            parsed_lattice[matched.group(1)]["type"] = matched.group(2).lower()
            
            if matched.group(2).lower() == "line":
                parsed_lattice[matched.group(1)]["elements"] = matched.group(3).split(",")
            else:

                element_data = dict([tuple(kv.split("=")) 
                                        for kv 
                                        in matched.group(3).split(",")])

                for key in element_data:
                    if element_data[key].lower() == "true":
                        element_data[key] == True
                    else:
                        element_data[key] = ast.literal_eval(element_data[key])

                parsed_lattice[matched.group(1)].update(element_data)

        else:
            print(fname+":", "Unable to parse line :\"", element.replace(" ",""), "\"")
            exit(-1)

    
    return parsed_lattice


def parse_lattice_toml(fname):
    '''
        Parse lattice in TOML format
    '''
    return toml.load(fname)

def parse_lattice_json(fname):
    '''
        Parse lattice in JSON format
    '''
    return toml.load(fname)

def dump_lattice_native(lattice):
    '''
        Emit a lattice in Genesis1.3v4 format
    '''
    emitted = ""
    for name, values in lattice.items():
        table = ""
        for key in values if key != "type":
            table += " " + key + " = " + str(values[key]) + ","

        line = name + ": " + values["type"].upper() + " = {" + table + "};\n"
        emitted += line

    return emitted

def validate_lattice(lattice, fname):
    '''
        Check wheter the elements in a lattice are valid
    '''
    for (element_name, lattice_element) in lattice.items():
        if lattice_element["type"] == "line":
            for contained in lattice_element["elements"]:
                if contained not in lattice:
                    print(fname+":",
                          "Lattice element: \"", contained, 
                          "\" contained in: \"", element_name, 
                          "\" not found")

if __name__ == "__main__":
    print(dump_lattice_native(parse_lattice_native(sys.argv[1]))
