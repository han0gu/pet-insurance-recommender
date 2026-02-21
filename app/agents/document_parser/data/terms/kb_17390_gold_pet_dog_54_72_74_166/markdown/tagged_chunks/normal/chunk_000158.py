from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 따른 해약환급금을 지급합니다.\n'
 '- \uf000 회사는 자동대출납입이 종료된 날부터 15일 이내에 자동대출납입이 종료되었음을\n'
 '- 서면, 전화(음성녹음) 또는 전자문서(SMS 포함) 등으로 계약자에게 안내하여 드립\n'
 '- 니다.\n'
 '# 용 어 풀 이 자동대출납입| 보험료를 제때에 납입하기 | 곤란한 경우에 | 계약자가 자동대출납입을 신청하면 |\n'
 '| --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 206,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
