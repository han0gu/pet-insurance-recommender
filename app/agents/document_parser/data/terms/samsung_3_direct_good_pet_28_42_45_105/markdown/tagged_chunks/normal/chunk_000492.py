from langchain_core.documents import Document

chunk = Document(
    page_content=('- 및 이들 사항의 증인이 있을 경우 그 주소와 성명\n'
 '- 2. 피해자로부터 손해배상청구를 받았을 경우\n'
 '- 3. 피해자로부터 손해배상책임에 관한 소송을 제기받았을 경우\n'
 '② 계약자 또는 피보험자가 제1항 각 호의 통지를 게을리하여 손해가 증가된 때에는 회\n'
 '사는 그 증가된 손해를 보상하지 않으며, 제1항 제3호의 통지를 게을리 한 때에는 소\n'
 '송비용과 변호사비용도 보상하지 않습니다. 다만, 계약자 또는 피보험자가 상법 제\n'
 '657조 제1항에 의해 보험사고의 발생을 회사에 알린 경우에는 제3조(보상하는 손해)'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000492',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
