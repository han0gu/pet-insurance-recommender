from langchain_core.documents import Document

chunk = Document(
    page_content=('가) "치매" 라 함은 정상적으로 성숙한 뇌가 질병이나 외상 후 기질성 손상으 로 파괴되어 한번 획득한 지적기능이 지속적 또는 전반적으로 '
 '저하되는 것 을 말한다. 나) 치매의 장해평가는 임상적인 증상 뿐 아니라 뇌영상검사(CT 및 MRI, SPECT 등)를 기초로 '
 '진단되어져야 하며, 18개월 이상 지속적인 치료 후 평가한다. 다만, 진단시점에 이미 극심한 치매 또는 심한 치매로 진행된 경 우에는 '
 '6개월간 지속적인 치료 후 평가한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 148},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000980',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
