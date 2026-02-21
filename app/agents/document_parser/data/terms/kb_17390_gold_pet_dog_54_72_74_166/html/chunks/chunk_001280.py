from langchain_core.documents import Document

chunk = Document(
    page_content=("위탁비용이</p><br><figure id='115' data-category='chart'><img "
 'data-coord="top-left:(824,772); bottom-right:(1431,958)" '
 '/><figcaption><p>Chart Type: '
 'bar</p></figcaption><table><thead><tr><td></td><td>보호</td><td>분쟁세외</td><td>부장</td></tr></thead><tbody><tr><td>item_01</td><td>180Not'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
