from langchain_core.documents import Document

chunk = Document(
    page_content=('근위지관절(제1지관절)\n'
 '(Pipjoint)\n'
 '중간설상골 외측설상골 (Lataral\n'
 '입방골(Cuboid) 증족지관절\n'
 '주상골\n'
 'Navicular (Mpjoint)\n'
 '리스프랑 관절'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
