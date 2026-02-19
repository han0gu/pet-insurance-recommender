from langchain_core.documents import Document

chunk = Document(
    page_content=('제17조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)\n'
 '\uf000 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 않아 보험료 납입이 연체 중인 경우에 회사는 14일(보험기 간이 1년 '
 '미만인 경우에는 7일) 이상의 기간을 납입최고(독 촉)기간(납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 '
 '다음 날까지로 합니다)으로 정 하여 아래 사항에 대하여 서면(등기우편 등), 전화(음성녹 음) 또는 전자문서 등으로 알려드립니다. 다만, '
 '해지 전에 발생한 보험금 지급사유에 대하여 회사는 보상합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000257',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
