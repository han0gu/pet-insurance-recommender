from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제10차 개정 이후 이 약관에서 보장하는 천식지속상태 해당여부는 피보험자가<br>진단된 당시 시행되고 있는 한국표준질병․사인분류에 '
 "따라 판단합니다.</p><br><p id='98' data-category='paragraph' "
 "style='font-size:14px'>162 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='99' "
 "data-category='paragraph' style='font-size:14px'>3"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001778',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
