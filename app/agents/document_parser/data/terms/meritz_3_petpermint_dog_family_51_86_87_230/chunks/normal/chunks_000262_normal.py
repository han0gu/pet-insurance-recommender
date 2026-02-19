from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자에게 전자적 상품설명장치를 활용하여 제1항에 따른 납입최고(독촉) 등을 한다는 사실을 미리 안내하 고 동의를 받을 것 ② 전자적 '
 '상품설명장치를 활용하여 안내한 납입최고(독 촉) 등을 계약자가 모두 수신하고 이해하였음을 확인 할 것 ③ 계약자가 질의를 하거나 추가적인 '
 '설명을 요청하는 등 전자적 상품설명장치의 활용을 중단할 것을 요구하는 경우, 회사는 전화 (음성녹음) 방법으로 전환하여 제1 항에 따른 '
 '납입최고(독촉) 등을 실시할 것 ④ 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것 ⑤ 제3호 및'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 105},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000262',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
