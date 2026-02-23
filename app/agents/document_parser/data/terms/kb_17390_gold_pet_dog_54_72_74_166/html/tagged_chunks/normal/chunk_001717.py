from langchain_core.documents import Document

chunk = Document(
    page_content=("이 약관에서 보장하는 상병 해당 여부를 다시 판단하지 않습니다.</p><p id='48' data-category='paragraph' "
 "style='font-size:14px'>158 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><h1 id='49' "
 "style='font-size:14px'>별표6 골절분류표<br>\uf000 약관에 규정하는 골절로 분류되는 상병은 제9차 개정 "
 '한국표준질병․사인분류(KCD,<br>통계청 고시 제2025-299호, 2026.1.1'),
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
 'indexing': {'chunk_id': 'chunk_001717',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
