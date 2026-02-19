from langchain_core.documents import Document

chunk = Document(
    page_content=('② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제11조 (계약 전 알릴 의무)\n'
 '계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건강진단할 때를 말합니다) 청약 서에서 질문한 사항에 대하여 알고 있는 사실을 '
 '반드시 사실대로 알려야(이하「계약 전 알릴 의무」라 하며, 상법상 「고지의무」와 같습니다) 합니다.\n'
 '<관련법규>\n'
 '[상법 제651조(고지의무위반으로 인한 계약해지)]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 71},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000384',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
