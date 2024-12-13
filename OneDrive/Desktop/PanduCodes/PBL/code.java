import java.util.*;

class Person {
    private String username;
    private String name;
    private int age;
    private String birthday;
    private List<String> hobbies;
    private List<String> interests;
    private String phoneNumber;

    public Person(String username, String name, int age, String birthday, List<String> hobbies, List<String> interests, String phoneNumber) {
        this.username = username;
        this.name = name;
        this.age = age;
        this.birthday = birthday;
        this.hobbies = hobbies;
        this.interests = interests;
        this.phoneNumber = phoneNumber;
    }

    // Getters for person details
    public String getUsername() {
        return username;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public String getBirthday() {
        return birthday;
    }

    public List<String> getHobbies() {
        return hobbies;
    }

    public List<String> getInterests() {
        return interests;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }
}

class Graph {
    private Map<String, Person> accounts;
    private Map<String, List<String>> connections;
    private Map<String, Set<String>> sharedHobbiesMap;
    private Map<String, Set<String>> sharedInterestsMap;

    public Graph() {
        this.accounts = new HashMap<>();
        this.connections = new HashMap<>();
        this.sharedHobbiesMap = new HashMap<>();
        this.sharedInterestsMap = new HashMap<>();
    }

    public void addAccount(Person person) {
        accounts.put(person.getUsername(), person);
        connections.put(person.getUsername(), new ArrayList<>()); // Initialize connections for the new account
        updateSharedLinks(person);
    }

    public Person getAccount(String username) {
        return accounts.get(username);
    }

    public List<String> getAllUsernames() {
        return new ArrayList<>(accounts.keySet());
    }

    public List<String> getConnections(String username) {
        return connections.get(username);
    }

    public void addConnection(String user1, String user2) {
        if (connections.containsKey(user1) && connections.containsKey(user2)) { // Check if both users exist in connections map
            connections.get(user1).add(user2);
            connections.get(user2).add(user1);
        }
    }

    public List<String> findMutuals(String user1, String user2) {
        List<String> mutuals = new ArrayList<>();
        List<String> friends1 = connections.getOrDefault(user1, new ArrayList<>()); // Check if user1 exists in connections map
        List<String> friends2 = connections.getOrDefault(user2, new ArrayList<>()); // Check if user2 exists in connections map

        for (String friend : friends1) {
            if (friends2.contains(friend)) {
                mutuals.add(friend);
            }
        }

        return mutuals;
    }

    private void updateSharedLinks(Person newPerson) {
        for (String username : accounts.keySet()) {
            if (!username.equals(newPerson.getUsername())) {
                Person person = accounts.get(username);
                if (person != null) {
                    checkAndUpdateSharedHobbies(newPerson, person);
                    checkAndUpdateSharedInterests(newPerson, person);
                }
            }
        }
    }

    private void checkAndUpdateSharedHobbies(Person newPerson, Person existingPerson) {
        if (newPerson != null && existingPerson != null) { // Check if both persons are not null
            Set<String> newHobbiesSet = new HashSet<>(newPerson.getHobbies());
            Set<String> existingHobbiesSet = new HashSet<>(existingPerson.getHobbies());
            newHobbiesSet.retainAll(existingHobbiesSet);
            if (!newHobbiesSet.isEmpty()) {
                sharedHobbiesMap.computeIfAbsent(newPerson.getUsername(), k -> new HashSet<>()).addAll(existingHobbiesSet);
                sharedHobbiesMap.computeIfAbsent(existingPerson.getUsername(), k -> new HashSet<>()).addAll(newHobbiesSet);
                addConnection(newPerson.getUsername(), existingPerson.getUsername());
            }
        }
    }

    private void checkAndUpdateSharedInterests(Person newPerson, Person existingPerson) {
        if (newPerson != null && existingPerson != null) { // Check if both persons are not null
            Set<String> newInterestsSet = new HashSet<>(newPerson.getInterests());
            Set<String> existingInterestsSet = new HashSet<>(existingPerson.getInterests());
            newInterestsSet.retainAll(existingInterestsSet);
            if (!newInterestsSet.isEmpty()) {
                sharedInterestsMap.computeIfAbsent(newPerson.getUsername(), k -> new HashSet<>()).addAll(existingInterestsSet);
                sharedInterestsMap.computeIfAbsent(existingPerson.getUsername(), k -> new HashSet<>()).addAll(newInterestsSet);
                addConnection(newPerson.getUsername(), existingPerson.getUsername());
            }
        }
    }
}

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        Graph graph = new Graph();

        // Sample user accounts
        createSampleAccounts(graph);

        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1. Create an account");
            System.out.println("2. See a person's account");
            System.out.println("3. Check mutuals between two people");
            System.out.println("4. View all existing accounts");
            System.out.println("5. Exit");
            System.out.print("Enter your choice: ");
            int choice = scanner.nextInt();
            scanner.nextLine(); // Consume newline

            switch (choice) {
                case 1:
                    createAccount(graph, scanner);
                    break;
                case 2:
                    viewAccount(graph, scanner);
                    break;
                case 3:
                    checkMutuals(graph, scanner);
                    break;
                case 4:
                    viewAllAccounts(graph);
                    break;
                case 5:
                    System.out.println("Exiting...");
                    System.exit(0);
                    break;
                default:
                    System.out.println("Invalid choice. Please enter a number between 1 and 5.");
            }
        }
    }

    private static void createSampleAccounts(Graph graph) {
        // Create sample accounts (50 in number)
        for (int i = 0; i < 50; i++) {
            String username = "user" + i;
            String name = "User " + i;
            int age = 20 + i; // Just a sample age
            String birthday = "January 1, 2000"; // Just a sample birthday
            List<String> hobbies = Arrays.asList("Reading", "Gaming"); // Just sample hobbies
            List<String> interests = Arrays.asList("Programming", "Music"); // Just sample interests
            String phoneNumber = "123-456-7890"; // Just a sample phone number

            Person person = new Person(username, name, age, birthday, hobbies, interests, phoneNumber);
            graph.addAccount(person);
        }
    }

    private static void createAccount(Graph graph, Scanner scanner) {
        System.out.println("\nEnter account details:");
        System.out.print("Username: ");
        String username = scanner.nextLine();
        System.out.print("Name: ");
        String name = scanner.nextLine();
        System.out.print("Age: ");
        int age = scanner.nextInt();
        scanner.nextLine(); // Consume newline
        System.out.print("Birthday: ");
        String birthday = scanner.nextLine();
        System.out.print("Hobbies (comma-separated list): ");
        String[] hobbiesArray = scanner.nextLine().split(",");
        List<String> hobbies = Arrays.asList(hobbiesArray);
        System.out.print("Interests (comma-separated list): ");
        String[] interestsArray = scanner.nextLine().split(",");
        List<String> interests = Arrays.asList(interestsArray);
        System.out.print("Phone Number: ");
        String phoneNumber = scanner.nextLine();

        Person person = new Person(username, name, age, birthday, hobbies, interests, phoneNumber);
        graph.addAccount(person);
        System.out.println("Account created successfully.");
    }

    private static void viewAccount(Graph graph, Scanner scanner) {
        System.out.print("\nEnter username to see account details: ");
        String viewUsername = scanner.nextLine();
        Person person = graph.getAccount(viewUsername);
        if (person != null) {
            System.out.println("Account Details for " + viewUsername + ":");
            System.out.println("Name: " + person.getName());
            System.out.println("Age: " + person.getAge());
            System.out.println("Birthday: " + person.getBirthday());
            System.out.println("Hobbies: " + person.getHobbies());
            System.out.println("Interests: " + person.getInterests());
            System.out.println("Phone Number: " + person.getPhoneNumber());
        } else {
            System.out.println("Account with username " + viewUsername + " not found.");
        }
    }

    private static void checkMutuals(Graph graph, Scanner scanner) {
        System.out.print("\nEnter usernames to check mutuals (comma-separated): ");
        String[] users = scanner.nextLine().split(",");
        if (users.length != 2) {
            System.out.println("Please provide exactly two usernames separated by comma.");
            return;
        }
        String user1 = users[0].trim();
        String user2 = users[1].trim();
        List<String> mutuals = graph.findMutuals(user1, user2);
        if (mutuals.isEmpty()) {
            System.out.println("No mutual connections found between " + user1 + " and " + user2);
        } else {
            int length = 0;
            for (String i : mutuals) {
                length ++;
            }
            System.out.println("Mutual connections between " + user1 + " and " + user2 + ": " );
            for(int m=0;m<length;m=m+7)
            {
                
                System.out.println(mutuals.get(m));
            }
        }
    }

    private static void viewAllAccounts(Graph graph) {
        List<String> allUsernames = graph.getAllUsernames();
        System.out.println("Existing accounts:");
        for (String username : allUsernames) {
            System.out.println(username);
        }
    }
}