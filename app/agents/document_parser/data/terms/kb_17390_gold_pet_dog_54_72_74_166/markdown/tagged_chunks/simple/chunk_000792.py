from langchain_core.documents import Document

chunk = Document(
    page_content=('- 적 의사표시로 제공한 경우, 계약자 또는 그 대리인이 보험계약 안내자료를 수신\n'
 '- 하였을 때에는 해당 문서를 드린 것으로 봅니다.\n'
 '- \uf000 계약자가 보험계약 안내자료에 대하여 전자적 방법의 수령을 원하지 않는 경우에\n'
 '- 는 청약한 날로부터 5영업일 이내에 보험계약 안내자료를 우편 등의 방법으로 계\n'
 '- 약자에게 드립니다.\n'
 '제4조(계약자의 알릴 의무)\uf000 계약자가- \n'
 '제3조(약관교부의 특례) 제1항에 정한 방법으로 보험계약 안내자료를 수령하고자 하는 경우 계약을 청약할 때 보험계약 안내자료를 수령할 '
 '전자우편(이메'),
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
 'indexing': {'chunk_id': 'chunk_000792',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
