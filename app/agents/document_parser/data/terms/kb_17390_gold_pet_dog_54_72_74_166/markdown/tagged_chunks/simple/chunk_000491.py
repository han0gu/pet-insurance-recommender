from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 1년)이 지났을 때\n'
 '- 3. 최초 계약을 체결한 날부터 3년이 지났을 때\n'
 '- 4. 회사가 이 계약을 청약할 때 반려동물의 건강상태를 판단할 수 있는 기초자료\n'
 '- (건강진단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어\n'
 '- 있는 사항으로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회\n'
 '- 사에 제출한 기초자료의 내용 중 중요사항을 고의로 사실과 다르게 작성한 때\n'
 '- 에는 계약을 해지할 수 있습니다)\n'
 '- 5. 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000491',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
