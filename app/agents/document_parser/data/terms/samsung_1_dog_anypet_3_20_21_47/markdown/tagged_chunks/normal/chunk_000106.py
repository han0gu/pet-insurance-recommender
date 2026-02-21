from langchain_core.documents import Document

chunk = Document(
    page_content=('- 그 권리를 지키거나 행사하기 위하여 지출한 필요 또는 유익하였던 비용\n'
 '- 다. 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 조정에 관한 비용\n'
 '- 라. 보험증권상의 보상한도액내의 금액에 대한 공탁보증보험료. 그러나 회사는 그러한 보증을\n'
 '- 제공할 책임은 부담하지 않습니다.\n'
 '- 마. 피보험자가 제8조(손해배상청구에 대한 회사의 해결) 제2항 및 제3항의 회사의 요구에 따르\n'
 '- 기 위하여 지출한 비용'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000106',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
