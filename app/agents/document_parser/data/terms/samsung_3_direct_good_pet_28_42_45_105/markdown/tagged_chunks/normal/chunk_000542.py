from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1회 보험료의 납입방법을 계약자의 지정 금융기관 지정계좌를 통한 자동납입으로\n'
 '- 가입하고자 하는 경우에, 보험회사(이하「회사」라 합니다)는 청약서를 접수하고 자동\n'
 '- 이체신청에 필요한 정보를 제공한 때(다만, 계약자의 귀책사유로 보험료 납입이 불가\n'
 '- 능한 경우에는 지정 금융기관 지정계좌로부터 제1회 보험료가 이체된 날을 기준으로\n'
 '- 합니다)를 청약일 및 제1회 보험료 납입일로 하여 보험계약(특별약관이 부가된 경우에\n'
 '- 는 특별약관을 포함합니다. 이하「보험계약」이라 합니다)「보험계약의 성립」의 규\n'
 '- 정을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000542',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
