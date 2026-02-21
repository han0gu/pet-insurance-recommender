from langchain_core.documents import Document

chunk = Document(
    page_content=('[장해지급률]\n'
 '질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로\n'
 '나타낸 것을 말합니다.④ 회사는 제1항의 규정에 정한 지급기일 내에 보험금을 지급하지 않았을 때(제2항의 규\n'
 '정에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급일까지'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000329',
              'chunk_char_len': 165,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
