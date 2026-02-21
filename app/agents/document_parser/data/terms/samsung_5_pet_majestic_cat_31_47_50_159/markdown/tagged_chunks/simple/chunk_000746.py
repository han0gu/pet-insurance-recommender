from langchain_core.documents import Document

chunk = Document(
    page_content=('사검사" 등을 추가실시 후 장해를 평가한다.- \n'
 '다. 귓바퀴의 결손- 137 -1) "굿바퀴의 대부분이 결손된 때" 라 함은 귓바퀴의 연골부가 1/2이상 결손된\n'
 '경우를 말한다.\n'
 '2) 귓바퀴의 연골부가 1/2 미만 결손이고 청력에 이상이 없으면 외모의 추상(추한\n'
 '모습)장해로만 평가한다.- \n'
 '# 라. 평형기능의 장해1) "평형기능에 장해를 남긴 때" 라 함은 전정기관 이상으로 보행 등 일상생활이\n'
 '어려운 상태로 아래의 평형장해 평가항목별 합산점수가 30점 이상인 경우를\n'
 '말한다.| 항목 | 내 용 | 점수 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000746',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
