from langchain_core.documents import Document

chunk = Document(
    page_content=('| item_01 | 4 | 1 |\n'
 '| item_02 | 2 | 1 |\n'
 '| item_03 | 2 | 2 |\n'
 '※ 설명\n'
 '보장개시일로부터 1년이내 발생한 슬관절탈구 : 보험금 미지급\uf000 제1항의「연간」이라 함은 계약일부터 매 1년 단위로 도\n'
 '래하는 계약해당일 전일까지의 기간을 말합니다.\n'
 '\uf000 반려동물이 제1항의 질병 또는 상해로 치료를 받던 중에119보험기간이 만료된 경우에도 만료일부터 180일 이내의 치료\n'
 '비는 제2항에 따라 보상하여 드립니다. 다만, 사고일 또는\n'
 '발병일부터 365일 이내인 경우에 한합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000267',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
