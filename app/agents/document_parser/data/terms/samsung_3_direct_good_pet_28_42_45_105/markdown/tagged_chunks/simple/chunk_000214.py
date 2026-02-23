from langchain_core.documents import Document

chunk = Document(
    page_content=('다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으로 대신합\n'
 '니다.<용어풀이># [납입기일]계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합니다.# 제 28조 (보험료의 자동대출납입)① '
 '계약자는 제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)\n'
 '에 따른 보험료의 납입최고(독촉)기간이 지나기 전까지 회사가 정한 방법에 따라 보험\n'
 '료의 자동대출납입을 신청할 수 있으며, 이 경우 제36조(보험계약대출) 제1항에 따른'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000214',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
