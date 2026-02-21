from langchain_core.documents import Document

chunk = Document(
    page_content=('. 약<br>2) 1하지(다리와 발가락)의 후유장해 지급률은 원칙적으로 각각 합산하되, 관<br>지급률은 60% 한도로 한다.</p><p '
 "id='124' data-category='paragraph' style='font-size:16px'>10. 손가락의 "
 "장해</p><br><p id='125' data-category='list'></p><p id='126' "
 "data-category='paragraph' style='font-size:16px'>가"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001611',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
