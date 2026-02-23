from langchain_core.documents import Document

chunk = Document(
    page_content=('- 으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다. 계약자의 전\n'
 '- 자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회사\n'
 '- 는 전자문서가 수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으로 다시 알려\n'
 '- 33 -14조(상해보험계약 후 알릴 의무) 제4항 또는 제5항에 따라 보험금을 지급합니다.- ⑥ 제1항에도 불구하고 알릴 의무를 '
 '위반한 사실이 보험금 지급사유 발생에 영향을 미쳤\n'
 '- 음을 회사가 증명하지 못한 경우에는 제4항 및 제5항에 관계없이 약정한 보험금을 지'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000050',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
