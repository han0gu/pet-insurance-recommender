from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 상해: 보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 신체(의수, 의족, 의 안, 의치 등 신체보조장구는 제외하나, '
 '인공장기나 부분 의치 등 신체에 이식되어 그 기능을 대신할 경우는 포함합니다)에 입은 상해를 말합니다. 2. 장해: '
 '[별표2]장해분류표에서 정한 기준에 따른 장해상태를 말합니다. 3'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 45},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 173,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
