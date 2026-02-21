from langchain_core.documents import Document

chunk = Document(
    page_content=('주 ) 1. 감염병 예방 및 관리에 관한 법률([시행 2022. 1. 13.][법률 제17893호, 2021. 1. 12., 타법개\n'
 '정]) 제2조(정의) 중 "보장대상 법정감염병" 이외의 감염병은 보장하지 않습니다.\n'
 '2 . 향후 「감염병의 예방 및 관리에 관한 법률」등 관계법령이 개정되어 신규로 추가되는 법정감\n'
 '염병이 생기더라도 "보장대상 법정감염병"에서 나열한 감염병만 보장되며, 신규로 추가되는\n'
 '감염병은 보장하지 않습니다.\n'
 '3 . 향후 「감염병의 예방 및 관리에 관한 법률」등 관계법령에서 제외되는 감염병이 생기더라도'),
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
 'indexing': {'chunk_id': 'chunk_000897',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
