"""
Tests for core domain modeling components.
"""

import unittest
from quarum.core.code import Code
from quarum.core.relationship import CodeRelationship
from quarum.core.code_system import CodeSystem
from quarum.core.enums import CSLRelationshipType

class TestCode(unittest.TestCase):
    """Tests for the Code class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        code = Code("code1", "User", "A system user")
        self.assertEqual(code.id, "code1")
        self.assertEqual(code.name, "User")
        self.assertEqual(code.definition, "A system user")
        self.assertEqual(code.attributes, [])
        self.assertEqual(code.methods, [])
        self.assertFalse(code.is_interface)
        self.assertFalse(code.is_abstract)
        self.assertFalse(code.is_enumeration)
    
    def test_add_attribute(self):
        """Test adding attributes."""
        code = Code("code1", "User")
        code.add_attribute("name", "String", "+")
        code.add_attribute("age", "Integer", "-")
        
        self.assertEqual(len(code.attributes), 2)
        self.assertEqual(code.attributes[0]["name"], "name")
        self.assertEqual(code.attributes[0]["type"], "String")
        self.assertEqual(code.attributes[0]["visibility"], "+")
        self.assertEqual(code.attributes[1]["name"], "age")
        self.assertEqual(code.attributes[1]["type"], "Integer")
        self.assertEqual(code.attributes[1]["visibility"], "-")
    
    def test_add_method(self):
        """Test adding methods."""
        code = Code("code1", "User")
        code.add_method("getName", "(): String", "+")
        code.add_method("setAge", "(age: Integer): void", "-")
        
        self.assertEqual(len(code.methods), 2)
        self.assertEqual(code.methods[0]["name"], "getName")
        self.assertEqual(code.methods[0]["signature"], "(): String")
        self.assertEqual(code.methods[0]["visibility"], "+")
        self.assertEqual(code.methods[1]["name"], "setAge")
        self.assertEqual(code.methods[1]["signature"], "(age: Integer): void")
        self.assertEqual(code.methods[1]["visibility"], "-")
    
    def test_set_as_interface(self):
        """Test setting a code as interface."""
        code = Code("code1", "UserService")
        code.set_as_interface()
        
        self.assertTrue(code.is_interface)
        self.assertIn("interface", code.stereotypes)
    
    def test_set_as_abstract(self):
        """Test setting a code as abstract."""
        code = Code("code1", "BaseUser")
        code.set_as_abstract()
        
        self.assertTrue(code.is_abstract)
        self.assertIn("abstract", code.stereotypes)
    
    def test_add_enum_value(self):
        """Test adding enum values."""
        code = Code("code1", "UserType")
        code.set_as_enumeration()
        code.add_enum_value("ADMIN")
        code.add_enum_value("USER")
        
        self.assertTrue(code.is_enumeration)
        self.assertIn("enumeration", code.stereotypes)
        self.assertEqual(len(code.enum_values), 2)
        self.assertIn("ADMIN", code.enum_values)
        self.assertIn("USER", code.enum_values)


class TestCodeRelationship(unittest.TestCase):
    """Tests for the CodeRelationship class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        rel = CodeRelationship(
            "rel1", "code1", "code2", 
            CSLRelationshipType.ASSOCIATION,
            "manages"
        )
        
        self.assertEqual(rel.id, "rel1")
        self.assertEqual(rel.source_code_id, "code1")
        self.assertEqual(rel.target_code_id, "code2")
        self.assertEqual(rel.relationship_type, CSLRelationshipType.ASSOCIATION)
        self.assertEqual(rel.association_name, "manages")
        self.assertEqual(rel.multiplicity["source"], "1")
        self.assertEqual(rel.multiplicity["target"], "*")
    
    def test_set_multiplicity(self):
        """Test setting multiplicity."""
        rel = CodeRelationship(
            "rel1", "code1", "code2", 
            CSLRelationshipType.ASSOCIATION
        )
        
        rel.set_multiplicity("0..1", "1..*")
        
        self.assertEqual(rel.multiplicity["source"], "0..1")
        self.assertEqual(rel.multiplicity["target"], "1..*")
    
    def test_relationship_type_check(self):
        """Test relationship type check methods."""
        is_a_rel = CodeRelationship(
            "rel1", "code1", "code2", 
            CSLRelationshipType.IS_A
        )
        
        impl_rel = CodeRelationship(
            "rel2", "code1", "code2", 
            CSLRelationshipType.IMPLEMENTATION
        )
        
        comp_rel = CodeRelationship(
            "rel3", "code1", "code2", 
            CSLRelationshipType.IS_PART_OF
        )
        
        self.assertTrue(is_a_rel.is_inheritance())
        self.assertFalse(is_a_rel.is_implementation())
        self.assertFalse(is_a_rel.is_composition())
        
        self.assertFalse(impl_rel.is_inheritance())
        self.assertTrue(impl_rel.is_implementation())
        self.assertFalse(impl_rel.is_composition())
        
        self.assertFalse(comp_rel.is_inheritance())
        self.assertFalse(comp_rel.is_implementation())
        self.assertTrue(comp_rel.is_composition())


class TestCodeSystem(unittest.TestCase):
    """Tests for the CodeSystem class."""
    
    def setUp(self):
        """Set up test environment."""
        self.code_system = CodeSystem()
        
        # Add some codes
        self.user = Code("user1", "User", "A system user")
        self.user.add_attribute("name", "String", "+")
        self.user.add_method("getName", "(): String", "+")
        
        self.admin = Code("admin1", "Admin", "An administrator")
        self.admin.add_attribute("level", "Integer", "+")
        self.admin.add_method("getLevel", "(): Integer", "+")
        
        self.service = Code("service1", "UserService", "Service for user operations")
        self.service.set_as_interface()
        self.service.add_method("findUser", "(id: String): User", "+")
    
    def test_add_code(self):
        """Test adding codes."""
        self.code_system.add_code(self.user)
        self.code_system.add_code(self.admin)
        self.code_system.add_code(self.service)
        
        self.assertEqual(len(self.code_system.codes), 3)
        self.assertIn("user1", self.code_system.codes)
        self.assertIn("admin1", self.code_system.codes)
        self.assertIn("service1", self.code_system.codes)
    
    def test_add_relationship(self):
        """Test adding relationships."""
        # Add codes
        self.code_system.add_code(self.user)
        self.code_system.add_code(self.admin)
        self.code_system.add_code(self.service)
        
        # Create relationships
        inheritance = CodeRelationship(
            "rel1", "admin1", "user1", 
            CSLRelationshipType.IS_A
        )
        
        implementation = CodeRelationship(
            "rel2", "admin1", "service1", 
            CSLRelationshipType.IMPLEMENTATION
        )
        
        # Add relationships
        self.assertTrue(self.code_system.add_relationship(inheritance))
        self.assertTrue(self.code_system.add_relationship(implementation))
        
        # Check relationships
        self.assertEqual(len(self.code_system.relationships), 2)
        
        # Check outgoing relationships
        admin = self.code_system.codes["admin1"]
        self.assertEqual(len(admin.outgoing_relationships), 2)
        
        # Check incoming relationships
        user = self.code_system.codes["user1"]
        service = self.code_system.codes["service1"]
        self.assertEqual(len(user.incoming_relationships), 1)
        self.assertEqual(len(service.incoming_relationships), 1)
    
    def test_duplicate_relationship(self):
        """Test adding duplicate relationships."""
        # Add codes
        self.code_system.add_code(self.user)
        self.code_system.add_code(self.admin)
        
        # Create relationships
        rel1 = CodeRelationship(
            "rel1", "admin1", "user1", 
            CSLRelationshipType.IS_A
        )
        
        rel2 = CodeRelationship(
            "rel2", "admin1", "user1", 
            CSLRelationshipType.IS_A
        )
        
        # Add relationships
        self.assertTrue(self.code_system.add_relationship(rel1))
        self.assertFalse(self.code_system.add_relationship(rel2))
        
        # Check relationships
        self.assertEqual(len(self.code_system.relationships), 1)
    
    def test_get_code_by_name(self):
        """Test getting code by name."""
        # Add codes
        self.code_system.add_code(self.user)
        self.code_system.add_code(self.admin)
        
        # Get codes
        user = self.code_system.get_code_by_name("User")
        admin = self.code_system.get_code_by_name("Admin")
        nonexistent = self.code_system.get_code_by_name("NonExistent")
        
        # Check results
        self.assertIsNotNone(user)
        self.assertEqual(user.id, "user1")
        self.assertIsNotNone(admin)
        self.assertEqual(admin.id, "admin1")
        self.assertIsNone(nonexistent)


if __name__ == '__main__':
    unittest.main()