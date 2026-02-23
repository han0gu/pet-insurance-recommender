from langchain_core.documents import Document

chunk = Document(
    page_content=('사업을 양도하였을 때 계약으로 인하여 생긴 권리와 의무를 함께 양도 한 것으로 봅\n'
 '니다.# 제20조(사기에 의한계약)- \uf000 계약자, 피보험자 또는 이들의 대리인의 사기에 의하여 계약이 성립되었음을 회\n'
 '- 사가 증명하는 경우에는 계약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)\n'
 '- 에 계약을 취소할 수 있습니다.\n'
 '- \uf000 제1항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게\n'
 '- 돌려드립니다.'),
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
 'indexing': {'chunk_id': 'chunk_000709',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
