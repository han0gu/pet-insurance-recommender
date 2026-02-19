from langchain_core.documents import Document

chunk = Document(
    page_content=('12) "눈꺼풀에 뚜렷한 결손을 남긴 때" 에 해당하는 경우에는 추상(추한 모습)장 해를 포함하여 장해를 평가한 것으로 보고 추상(추한 '
 '모습)장해를 가산하지 않 는다. 다만, 안면부의 추상(추한 모습)은 두 가지 장해평가 방법 중 피보험자에 게 유리한 것을 적용한다.\n'
 '2. 귀의 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 두 귀의 청력을 완전히 잃었을 때 | 80\n'
 '2) 한 귀의 청력을 완전히 잃고, 다른 귀의 청력에 심한 장해를 남긴 때 | 45\n'
 '3) 한 귀의 청력을 완전히 잃었을 때 | 25'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye', 'other']},
 'indexing': {'chunk_id': 'chunk_000878',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
