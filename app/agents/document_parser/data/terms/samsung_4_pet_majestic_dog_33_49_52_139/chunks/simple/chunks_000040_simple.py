from langchain_core.documents import Document

chunk = Document(
    page_content=('해당월 마지막 날까지 1개월간 확정 적용합니다.\n'
 '<용어풀이>\n'
 '[공시이율]\n'
 '전통적인 보험상품에 적용되는 이율이 장기·고정금리이기 때문에 시중금리가 급격하게 변동할 경 우 이에 대응하지 못하는 점을 고려하여, '
 '시중의 지표금리 등에 연동하여 일정기간 마다 변동되는 이율을 말합니다.\n'
 '② 제1항의 공시이율은 이 보험의 사업방법서에서 정한 방법에 따라 회사의 운용자산이 익률과 외부지표금리를 가중평균하여 산출된 '
 '공시기준이율에서 향후 예상수익 등을 고려한 조정률을 가감하여 결정합니다.\n'
 '<용어풀이>\n'
 '[운용자산이익률]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
