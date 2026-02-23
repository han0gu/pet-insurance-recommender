from langchain_core.documents import Document

chunk = Document(
    page_content=('우 회사는 자동대출납입 신청내역을 서면, 전화(음성녹음) 또는 전자문서(SMS포함)\n'
 '등으로 계약자에게 알려 드립니다.<용어풀이># [자동대출납입]보험료를 제때에 납입하기 곤란한 경우에 계약자가 자동대출납입을 신청하면 해당 '
 '보험 상품의\n'
 '해약환급금 범위 내에서 납입할 보험료를 자동적으로 대출하여 이를 보험료 납입에 충당하는\n'
 '서비스를 말합니다.② 제1항의 규정에 의한 대출금과 보험료의 자동대출납입일의 다음날부터 그 다음 보험\n'
 '료의 납입최고(독촉)기간까지의 이자(보험계약대출이율 이내에서 회사가 별도로 정하'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000241',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
