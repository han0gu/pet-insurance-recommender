from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 제2조(보장특약의 자동갱신)에서 정한 갱신제한 사유 및 제2항의 갱신계약\n'
 '- 보험료에 대하여 갱신 전 보장특약의 보험기간이 끝나기 15일 전까지 그 내용을\n'
 '- 계약자에게 서면, 전화(음성녹음) 또는 전자문서 등으로 안내하여 드립니다.\n'
 '- \uf000 제1항에도 불구하고 법령의 제․개정, 금융위원회의 명령 또는 제도적인 변경에 따\n'
 '- 라 약관이 변경된 경우에는 갱신일 현재의 변경된 약관을 적용합니다.\n'
 '- \uf000 제4항에 따라 변경된 약관을 적용하게 되어 보장내용이 변경되는 경우, 회사는 제'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000778',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
