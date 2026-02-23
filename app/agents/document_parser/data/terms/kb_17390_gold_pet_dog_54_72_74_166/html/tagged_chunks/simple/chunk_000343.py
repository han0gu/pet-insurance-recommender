from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 장해분</p><br><p id='193' data-category='paragraph' "
 "style='font-size:14px'>72 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='194' "
 "data-category='paragraph' style='font-size:14px'>류표의 각 신체부위별 판정기준에서 별도로 정한 "
 '경우에는 그 기준에 따릅니다.<br>\uf000 이미 이 보장에서 후유장해보험금 지급사유에 해당되지 않았거나(보장개시 이전<br>의 '
 '원인에 의하거나 또는 그 이전에 발생한 후유장해를'),
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
 'indexing': {'chunk_id': 'chunk_000343',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
