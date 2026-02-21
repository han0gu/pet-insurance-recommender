from langchain_core.documents import Document

chunk = Document(
    page_content=('변경되더라도 이 약관에서 보장하는 질병 해당 여부를 다시 판단하지 않습니다. 법\n'
 'ㆍ규정별표16 특정정신질환 분류표- 163 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 163\uf000 약관에 규정하는 '
 '특정정신질환으로 분류되는 질병은 제9차 개정 한국표준질병․사'),
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
 'indexing': {'chunk_id': 'chunk_001024',
              'chunk_char_len': 146,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
