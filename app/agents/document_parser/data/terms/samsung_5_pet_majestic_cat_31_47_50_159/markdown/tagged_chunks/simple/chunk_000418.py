from langchain_core.documents import Document

chunk = Document(
    page_content=('- 험료 납입면제) 및 제6조(보험료 납입면제에 관한 세부규정)를 적용하지 않습니다.\n'
 '- ③ 제1항에도 불구하고, 특별약관 일반사항 제35조(해 약환급금)를 적용하지 않으며 보통\n'
 '- 약관 제36조(해약환급금)을 적용합니다.\n'
 '# 제2관 개별사항# 제1조 (보험금의 지급사유)- ① 회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 '
 '이라\n'
 '- 합니다) 중에 상해 또는 진단확정된 질병으로 「깁스(Cast)치료」 (이하 「깁스치료」 라'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000418',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
