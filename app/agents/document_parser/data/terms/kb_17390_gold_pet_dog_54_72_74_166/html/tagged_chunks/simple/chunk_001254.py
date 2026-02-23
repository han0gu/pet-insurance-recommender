from langchain_core.documents import Document

chunk = Document(
    page_content=('입원기간 중 의사의 지시를 따르지 않은 때에는 회사<br>는 반려동물 위탁비용의 전부 또는 일부를 지급하지 않습니다.</p><br><p '
 "id='73' data-category='list'></p><br><p id='74' data-category='paragraph' "
 "style='font-size:14px'>126 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='75' "
 "data-category='paragraph' style='font-size:14px'>\uf000 피보험자가 병원 또는 의원을 이전하여 "
 '입원한'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001254',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
