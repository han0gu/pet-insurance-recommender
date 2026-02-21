from langchain_core.documents import Document

chunk = Document(
    page_content=('상품 또는 금융상품자문에 관한 계약의 청약을 한<br>일반금융소비자는 다음 각 호의 구분에 따른 기간(거래 당사자 사이에 다음 '
 "각<br>호의 기간보다 긴 기간으로 약정한 경우에는 그 기간) 내에 청약을 철회할 수 있</p><br><h1 id='191' "
 "style='font-size:14px'>다.</h1><br><p id='192' data-category='paragraph' "
 "style='font-size:14px'>1"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000154',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
