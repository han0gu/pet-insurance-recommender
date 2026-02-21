from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건강진단할 때를 말합니다) "
 '청약<br>서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 「계약 전<br>알릴 의무」라 하며, '
 "상법상「고지의무」와 같습니다) 합니다.</p><br><h1 id='103' style='font-size:14px'>【계약 전 알릴 "
 "의무】</h1><br><p id='104' data-category='paragraph' style='font-size:14px'>상법 "
 '제651조에서 정하고'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000081',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
