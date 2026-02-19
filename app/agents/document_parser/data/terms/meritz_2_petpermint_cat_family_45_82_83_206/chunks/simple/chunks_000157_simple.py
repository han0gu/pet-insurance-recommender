from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 중도인출금은 계약자가 요청한 시점에서 계산된 기본계약 해약환급금과 기본계약 적립부분 해약환급금 중 적은 금액의 '
 '80% 범위 내에서 신청할 수 있습니다. 중도인 출금의 총 누적액(중도인출 원금과 이자의 합계액을 말합니 다)은 중도인출금을 한번도 '
 '지급하지 않았을 경우의 기본계 약 해약환급금과 기본계약 적립부분 해약환급금 중 적은 금 액의 80%를 한도로 합니다. 다만, 이 계약에서 '
 '정한 보험계 약대출금이 있는 때에는 그 원금과 이자의 합계액을 공제한 후의 잔액을 기준으로 합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 78},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000157',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
