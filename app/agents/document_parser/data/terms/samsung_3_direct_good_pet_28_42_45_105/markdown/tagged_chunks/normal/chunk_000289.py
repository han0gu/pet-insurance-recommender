from langchain_core.documents import Document

chunk = Document(
    page_content=('할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합\n'
 '니다.# ③ 지급금과 이자율 관련 용어1. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를\n'
 '원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.<예시안내>[연단위 복리]'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000289',
              'chunk_char_len': 169,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
