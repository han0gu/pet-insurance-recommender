from langchain_core.documents import Document

chunk = Document(
    page_content=("보장하는 질병 해당 여부를 다시 판단하지 않습니다.</p><table id='84' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>별표12</td><td>6대호흡계특정질환 "
 '분류표</td><td>법</td></tr><tr><td>\uf000 약관에 규정하는</td><td>6대호흡계특정질환으로 '
 "질병은</td><td>분류되는 제9차 개정 한국표준질</td></tr></tbody></table><br><p id='85' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001764',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
