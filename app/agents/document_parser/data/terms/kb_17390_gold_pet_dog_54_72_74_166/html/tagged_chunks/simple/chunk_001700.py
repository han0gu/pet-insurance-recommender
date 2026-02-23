from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제10차 개정 이후 이 약관에서 보장하는 외모특정상해 해당여부는 피보험자가</p><br><p id='31' "
 "data-category='paragraph' style='font-size:16px'>진단된 당시 시행되고 있는 한국표준질병․사인분류에 "
 "따라 판단합니다.</p><br><p id='32' data-category='paragraph' "
 "style='font-size:16px'>4"),
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
 'indexing': {'chunk_id': 'chunk_001700',
              'chunk_char_len': 217,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
