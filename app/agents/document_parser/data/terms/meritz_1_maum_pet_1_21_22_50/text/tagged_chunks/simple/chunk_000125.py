from langchain_core.documents import Document

chunk = Document(
    page_content=('14. 가입 반려견의 소음, 냄새, 털날림으로 인하여 발생한 배상책임\n'
 '15. 가입 반려견이 질병을 전염시켜 발생한 배상책임- 23 -제5조(손해의 통지 및 조사)① 계약자 또는 피보험자는 아래와 같은 사실이 '
 '있는 경우에는 지체없이 그 내용을 회사에\n'
 '알려야 합니다.1. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고상황\n'
 '및 이들 사항의 증인이 있을 경우 그 주소와 성명\n'
 '2. 피해자로부터 손해배상청구를 받았을 경우'),
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
 'indexing': {'chunk_id': 'chunk_000125',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
