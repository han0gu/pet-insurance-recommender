from langchain_core.documents import Document

chunk = Document(
    page_content=('제 4조 (이물제거(내시경) 및 이물제거(구토유발약물)의 정의)\n'
 '① 이 특별약관에서「이물제거(내시경)」란 반려견의 위장 등 내부의 이물질을 제거하기 위하여 수술을 동반하지 않고 내시경 및 내시경포셉을 '
 '이용하여 비침습적으로 시행하 는 의료행위를 말합니다. ② 이 특별약관에서「이물제거(구토유발약물)」란 반려견의 위장 등 내부의 이물질을 제 '
 '거하기 위하여 수술 또는 내시경을 동반하지 않고 구토유발을 목적으로 한 약물을 이 용한 의료행위를 말합니다.\n'
 '<유의사항>\n'
 '[수술]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000478',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
