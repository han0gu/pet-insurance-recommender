from langchain_core.documents import Document

chunk = Document(
    page_content=('책임을 지는 것과 다름), 어느 1인의 이행으로 나머지 사람들도 책임을 면하게 되는 것을 말합니\n'
 '다.# 제3관 계약자의 계약 전 알릴 의무 등# 제13조 (계약 전 알릴 의무)계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 '
 '건강진단할 때를 말합니다) 청약\n'
 '서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 「계약 전\n'
 '알릴 의무」 라 하며, 상법상 「고지의무」 와 같습니다) 합니다. 다만, 진단계약의 경우 의\n'
 '료법 제3조(의료기관)의 규정에 따른 종합병원과 병원에서 직장 또는 개인이 실시한 건강'),
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
 'indexing': {'chunk_id': 'chunk_000034',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
