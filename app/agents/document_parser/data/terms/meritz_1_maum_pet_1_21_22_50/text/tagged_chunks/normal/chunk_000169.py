from langchain_core.documents import Document

chunk = Document(
    page_content=('거래은행의 지정계좌로부터 제1회 보험료가 이체된 날을 기준으로 합니다)를 청약일 및\n'
 '제1회 보험료 납입일로 하여 보통약관의 제19조(보험계약의 성립)의 규정을 적용합니\n'
 '다.\n'
 '② 제1항의 경우에 회사는 청약서를 접수한 날로부터 30일 이내에 승낙 또는 거절하여야\n'
 '하며, 승낙한 때에는 금융기관의 해당계좌에서 제1회 보험료를 받고 보험증권을 드립니\n'
 '다.제2조(계약 후의 알릴의무)계약자는 지정계좌의 번호가 변경 또는 거래 정지된 경우에는 이 사실을 즉시 회사에 알려'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000169',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
