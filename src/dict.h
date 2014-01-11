/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL v2 license http://www.gnu.org/licenses/gpl.html .

*/

#ifndef __DICT_H__
#define __DICT_H__

#include "fArray.h"

#define hashPrime 521

inline unsigned int hashNumber( int i ) { return (unsigned int)i; }
inline unsigned int hashNumber( unsigned int i ) { return i; }

template<class Key, class Value>
class dictionaryIterator;

template<class Key, class Value>
class dictionary
{

public:

    struct pair {
        Key key;
        Value value;
    };

    dictionary( const Value &defaultValue_ ) : 
        defaultValue( defaultValue_ ), 
        table( hashPrime ) {}

	// default value をセットする。
	void setDefaultValue( const Value &defaultValue_ ) { defaultValue = defaultValue_; }

    // 値渡しによってデータを返す。
    Value operator[] (const Key &key) const;

    // 参照渡しによってデータを返す。
    const Value &operator() (const Key &key) const;
    Value &operator() (const Key &key);

    // データへの参照を返す。
    Value *pData(const Key &key);

    void clear() { table.clear( fArray<pair>() ); }

    // iterator 関係
    void toHead( dictionaryIterator<Key,Value> &iter ) const
    {
        iter.i = 0;
        iter.j = 0;
        iter.isValid_ = true;
        if( table[iter.i].length() == 0 )
            toNext(iter);
    }

    void toNext( dictionaryIterator<Key,Value> &iter ) const
    {
        ++(iter.j);
        if( iter.j < table[iter.i].length() )
            return;
        iter.j = 0;
        while( 1 )
        {
            ++(iter.i);
            if( hashPrime <= iter.i )
            {
                iter.isValid_ = false;
                return;
            }
            if( 0 < table[iter.i].length() )
                break;
        }
    }

    const Key *getKey( dictionaryIterator<Key,Value> &iter ) const
    {
        if( iter.isValid() )
            return &(table[iter.i][iter.j].key);
        return 0;
    }

    const Value *getValue( dictionaryIterator<Key,Value> &iter ) const
    {
        if( iter.isValid() )
            return &(table[iter.i][iter.j].value);
        return 0;
    }

private:
    
    Value defaultValue;
    fArray<fArray<pair> > table;
};

template<class Key, class Value>
class dictionaryIterator
{
    friend class dictionary<Key, Value>;

public:
    dictionaryIterator() : isValid_(false) {}
    bool isValid() { return isValid_; }

protected:
    unsigned int i;
    unsigned int j;
    bool isValid_;
};

// 値渡しによってデータを返す。
template<class Key, class Value>
Value dictionary<Key, Value>::operator [] (const Key &key) const 
{
    const fArray<pair> &candidates  = table[(unsigned int)hashNumber(key) % hashPrime];

    for( unsigned int i = 0; i < candidates.length(); ++i )    
    {
        if( candidates[i].key == key )
            return candidates[i].value;
    }

    return defaultValue;
}

// 参照渡しによってデータを返す。
// const version.
template<class Key, class Value>
const Value &dictionary<Key, Value>::operator () (const Key &key) const 
{
    fArray<pair> &candidates  = table[(unsigned int)hashNumber(key) % hashPrime];

    for( int i = 0; i < candidates.length(); ++i )    
    {
        if( candidates[i].key == key )
            return candidates[i].value;
    }

    return defaultValue;
}

// 参照渡しによってデータを返す。
// non-const version.
template<class Key, class Value>
Value &dictionary<Key, Value>::operator () (const Key &key)
{
    fArray<pair> &candidates  = table[(unsigned int)hashNumber(key) % hashPrime];
	
	unsigned int i;
    for( i = 0; i < candidates.length(); ++i )    
    {
        if( candidates[i].key == key )
            return candidates[i].value;
    }

    pair item;
    item.key = key;
    item.value = defaultValue;

    candidates.push_back( item );

    return candidates[i].value;
}

// データへの参照を返す。
template<class Key, class Value>
Value *dictionary<Key, Value>::pData(const Key &key)
{
    fArray<pair> &candidates  = table[(unsigned int)hashNumber(key) % hashPrime];

    for( unsigned int i = 0; i < candidates.length(); ++i )    
    {
        if( candidates[i].key == key )
            return &( candidates[i].value );
    }

    pair item;
    item.key = key;
    item.value = defaultValue;

    candidates.push_back( item );

    return &( candidates[i].value );
}

#endif

/*
#ifndef __DICT_H__
#define __DICT_H__

#include "fArray.h"

#define hashPrime 521

template<class Key, class Value>
class dictionary
{

public:

    struct pair {
        Key key;
        Value value;
    };

    dictionary( const Value &defaultValue_ ) : 
        defaultValue( defaultValue_ ), 
        table( hashPrime ) {}

    // 値渡しによってデータを返す。
    Value operator[] (const Key &key) const;

    // 参照渡しによってデータを返す。
    const Value &operator() (const Key &key) const;
    Value &operator() (const Key &key);

    // データへの参照を返す。
    Value *pData(const Key &key);

    void clear() { table.clear( fArray<pair>() ); }

private:
    
    Value defaultValue;
    fArray<fArray<pair> > table;

};

// 値渡しによってデータを返す。
template<class Key, class Value>
Value dictionary<Key, Value>::operator [] (const Key &key) const 
{
    const fArray<pair> &candidates  = table[(unsigned int)hashNumber(key) % hashPrime];

    for( unsigned int i = 0; i < candidates.length(); ++i )    
    {
        if( candidates[i].key == key )
            return candidates[i].value;
    }

    return defaultValue;
}

// 参照渡しによってデータを返す。
// const version.
template<class Key, class Value>
const Value &dictionary<Key, Value>::operator () (const Key &key) const 
{
    fArray<pair> &candidates  = table[(unsigned int)hashNumber(key) % hashPrime];

    for( int i = 0; i < candidates.length(); ++i )    
    {
        if( candidates[i].key == key )
            return candidates[i].value;
    }

    return defaultValue;
}

// 参照渡しによってデータを返す。
// non-const version.
template<class Key, class Value>
Value &dictionary<Key, Value>::operator () (const Key &key)
{
    fArray<pair> &candidates  = table[(unsigned int)hashNumber(key) % hashPrime];

    for( unsigned int i = 0; i < candidates.length(); ++i )    
    {
        if( candidates[i].key == key )
            return candidates[i].value;
    }

    pair item;
    item.key = key;
    item.value = defaultValue;

    candidates.push_back( item );

    return candidates[i].value;
}

// データへの参照を返す。
template<class Key, class Value>
Value *dictionary<Key, Value>::pData(const Key &key)
{
    fArray<pair> &candidates  = table[(unsigned int)hashNumber(key) % hashPrime];

    for( unsigned int i = 0; i < candidates.length(); ++i )    
    {
        if( candidates[i].key == key )
            return &( candidates[i].value );
    }

    pair item;
    item.key = key;
    item.value = defaultValue;

    candidates.push_back( item );

    return &( candidates[i].value );
}

#endif
*/
