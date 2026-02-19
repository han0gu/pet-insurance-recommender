from langchain_core.documents import Document

chunk = Document(
    page_content=('[별표-질병관련1] 특정법정감염병 분류표\n'
 '약 관에 규정하는 특정법정감염병으로 분류되는 질병은「감염병의 예방 및 관리에 관한 법률[시행 2022. 1. 13.][법률 '
 '제17893호, 2021. 1. 12., 타법개정]」제 2 조(정의)에 해 당하는 질병 중 다음에 적은 질병을 말합니다.\n'
 '보장대상 법정감염병\n'
 '1 . 콜레라 | 15. 말라리아\n'
 '2 . 장티푸스 | 16. 성홍열\n'
 '3 . 파라티푸스 | 17. | 수막구균감염증\n'
 '4 . 세균성이질 | 18. 레지오넬라증\n'
 '5 . 장출혈성대장균감염증 | 19. | 비브리오패혈증'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 158},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001031',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
