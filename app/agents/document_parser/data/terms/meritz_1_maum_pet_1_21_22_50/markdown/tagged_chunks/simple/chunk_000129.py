from langchain_core.documents import Document

chunk = Document(
    page_content=('- 12. 피보험자와 세대를 같이하는 친족에 대한 배상책임\n'
 '- 13. 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으로 이용하는\n'
 '- 중에 발생한 손해에 대한 배상책임\n'
 '- 14. 가입 반려견의 소음, 냄새, 털날림으로 인하여 발생한 배상책임\n'
 '- 15. 가입 반려견이 질병을 전염시켜 발생한 배상책임\n'
 '- 23 -# 제5조(손해의 통지 및 조사)① 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우에는 지체없이 그 내용을 회사에'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000129',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
