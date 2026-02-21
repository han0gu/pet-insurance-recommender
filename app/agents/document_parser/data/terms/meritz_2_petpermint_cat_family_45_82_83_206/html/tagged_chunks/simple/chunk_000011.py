from langchain_core.documents import Document

chunk = Document(
    page_content=("지급금과 이자율 관련 용어</p><br><table id='15' "
 "style='font-size:16px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>연단위 "
 '복리</td><td>회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로 하는 '
 '이자 계산방법을 말합니다.</td></tr><tr><td>평균공시 이율</td><td>전체 보험회사 공시이율의 평균으로, 이 계약 체결 '
 '시점의 이율을 말합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000011',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
