from langchain_core.documents import Document

chunk = Document(
    page_content=('신고여부와 상관없이 “특정법정감염병”의 보장대상에서는 제외되지 않습니다.- 158 -[별표-질병관련1.1] 감염병병원체 확인기관대 상이 '
 '되는 “주요법정감염병”의 감염병의 병원체를 확인할 수 있는 기관은「감염병의\n'
 '예방 및 관리에 관한 법률 」[시행 2022.1.13.][법률 제17893호, 2021. 1. 12,, 타법개\n'
 '정] 제16조의 2(감염병병원체 확인기관) 제1항에서 정한 기관을 말하며, 세부내용은 다\n'
 '음을 말합니다.다음 각 호의 기관(이하 "감염병병원체 확인기관"이라 한다)은 실험실 검사 등을 통하여 감염병병'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000898',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
