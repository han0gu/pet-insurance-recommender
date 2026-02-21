from langchain_core.documents import Document

chunk = Document(
    page_content=('을 말함<예시안내>[중도인출금의 한도 예시]- 36 -중도인출 시점에서 계산된 기본계약 해약환급금과 기본계약 적립부분 해약환급금 중 적은 '
 '금액이\n'
 '100만원인 경우- → 총 중도인출 가능액 = 100만원 × 80% = 80만원\n'
 '- → 기 신청한 보험계약대출금이 있는 경우(원금과 이자의 합계를 30만원으로 가정) 중도인출 가능액\n'
 '- ＝ 80만원(총 중도인출 가능액) － 30만원 ＝ 50만원\n'
 '# 제11조 (만기환급금의 지급)① 회사는 보험기간이 끝난 때에는 적립부분 순보험료에 대하여 보험료납입일(회사에 입'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000038',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
