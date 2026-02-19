from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자: 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 말합니다. 2. 보험수익자: 보험금 지급사유가 발생하는 때에 '
 '회사에 보험금을 청구하여 받을 수 있는 사람을 말합니다. 3. 보험증권: 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 '
 '드리는 증 서를 말합니다. 4. 진단계약: 계약을 체결하기 위하여 반려견이 건강진단을 받아야 하는 계약을 말합 니다. 5. 피보험자: '
 '반려견의 소유와 관련하여 보험사고로 손해를 입은 사람을 말합니다. 6'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 66},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000330',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
