from langchain_core.documents import Document

chunk = Document(
    page_content=('서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하「계약 전\n'
 '알릴 의무」라 하며, 상법상「고지의무」와 같습니다) 합니다. 다만, 진단계약의 경우 의\n'
 '료법 제3조(의료기관)의 규정에 따른 종합병원과 병원에서 직장 또는 개인이 실시한 건강\n'
 '진단서 사본 등 건강상태를 판단할 수 있는 자료로 건강진단을 대신할 수 있습니다.<관련법규>[상법 제651조(고지의무위반으로 인한 '
 '계약해지)]\n'
 '보험계약당시에 보험계약자 또는 피보험자가 고의 또는 중대한 과실로 인하여 중요한 사항을'),
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
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
