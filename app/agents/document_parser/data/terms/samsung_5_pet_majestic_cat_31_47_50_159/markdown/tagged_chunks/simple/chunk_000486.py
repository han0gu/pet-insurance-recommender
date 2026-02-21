from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 피보험자가 부담한 의료비 × | 다른 계약이 없는 것으로 하여 각각 계약의 지급보험금의 합계액 |\n'
 '② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한\n'
 '지급보험금 결정에는 영향을 미치지 않습니다.# 제11조 (계약 전 알릴 의무)계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 '
 '건강진단할 때를 말합니다) 청약- 100 -# 서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 「계약 전'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000486',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
